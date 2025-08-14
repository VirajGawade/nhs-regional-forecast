import argparse
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
args = parser.parse_args()

YEAR = args.year
MONTH = args.month
target_date = pd.Timestamp(year=YEAR, month=MONTH, day=1)
date_str = target_date.strftime("%Y-%m")

# Config paths
DATA_PATH = "regional_engineered_features.csv"
ML_MODEL_DIR = "regional_ml_monthly_models"
ML_SCALER_DIR = "regional_ml_monthly_scalers"
LSTM_MODEL_DIR = "."  # lstm_model_{region}.keras
LSTM_SCALER_DIR = "."  # scaler_{region}.save

SEQ_LEN = 12
ML_WEIGHT = 0.7
LSTM_WEIGHT = 0.3

# Load full data
df_all = pd.read_csv(DATA_PATH, parse_dates=["Month"])
df_all = df_all.sort_values("Month")

region_col = "Region_unified"
target_col = "Total_Attendances"

regions = sorted(df_all[region_col].dropna().unique())

results = []

for region in regions:
    print(f"\n[INFO] Processing region: {region}")

    # Filter data for region
    df = df_all[df_all[region_col] == region].copy()
    df = df.sort_values("Month").reset_index(drop=True)

    if len(df) < SEQ_LEN + 1:
        print(f"[WARNING] Not enough data for region {region}. Skipping...")
        continue

    # ML prediction
    ml_model_path_xgb = os.path.join(ML_MODEL_DIR, f"{region.replace(' ', '_')}_XGBoost_monthly.pkl")
    ml_model_path_rf = os.path.join(ML_MODEL_DIR, f"{region.replace(' ', '_')}_RandomForest_monthly.pkl")
    ml_scaler_path = os.path.join(ML_SCALER_DIR, f"{region.replace(' ', '_')}_scaler.save")

    if not (os.path.exists(ml_model_path_xgb) and os.path.exists(ml_model_path_rf) and os.path.exists(ml_scaler_path)):
        print(f"[WARNING] Missing ML model or scaler for region {region}, skipping ML prediction.")
        ml_pred = None
    else:
        xgb_model = joblib.load(ml_model_path_xgb)
        rf_model = joblib.load(ml_model_path_rf)
        scaler = joblib.load(ml_scaler_path)

        ml_features = df.drop(columns=[region_col, "Region", "Month", "Type", target_col], errors='ignore')
        ml_features = ml_features.select_dtypes(include=[np.number])

        input_features = ml_features.tail(SEQ_LEN).mean().to_frame().T
        input_features_scaled = scaler.transform(input_features)

        xgb_pred = xgb_model.predict(input_features_scaled)[0]
        rf_pred = rf_model.predict(input_features_scaled)[0]

        # Average RF and XGB predictions for ML prediction
        ml_pred = (xgb_pred + rf_pred) / 2

    # LSTM prediction
    lstm_model_path = os.path.join(LSTM_MODEL_DIR, f"lstm_model_{region}.keras")
    lstm_scaler_path = os.path.join(LSTM_SCALER_DIR, f"scaler_{region}.save")

    if not (os.path.exists(lstm_model_path) and os.path.exists(lstm_scaler_path)):
        print(f"[WARNING] Missing LSTM model or scaler for region {region}, skipping LSTM prediction.")
        lstm_pred = None
    else:
        lstm_model = load_model(lstm_model_path)
        lstm_scaler = joblib.load(lstm_scaler_path)

        numeric_cols = df.drop(columns=["Month", "Region", region_col, "Type"]).select_dtypes(include=[np.number])
        scaled_data = lstm_scaler.transform(numeric_cols)

        X_seq = []
        dates = []
        for i in range(len(df) - SEQ_LEN):
            seq = scaled_data[i:i + SEQ_LEN]
            X_seq.append(seq)
            dates.append(df.iloc[i + SEQ_LEN]["Month"])

        if target_date not in dates:
            print(f"[INFO] Forecasting future date for {region} - {date_str}")
            X_input = np.array([scaled_data[-SEQ_LEN:]])
        else:
            idx = dates.index(target_date)
            print(f"[INFO] Validating known date for {region} - {date_str}")
            X_input = np.array([X_seq[idx]])

        y_pred_scaled = lstm_model.predict(X_input, verbose=0)[0][0]
        last_frame = X_input[0, -1, :].copy()
        target_idx = numeric_cols.columns.get_loc(target_col)
        last_frame[target_idx] = y_pred_scaled

        inv_scaled = lstm_scaler.inverse_transform([last_frame])
        lstm_pred = inv_scaled[0][target_idx]

    # Combine predictions 
    if (ml_pred is not None) and (lstm_pred is not None):
        hybrid_pred = ML_WEIGHT * ml_pred + LSTM_WEIGHT * lstm_pred
    elif ml_pred is not None:
        hybrid_pred = ml_pred
    elif lstm_pred is not None:
        hybrid_pred = lstm_pred
    else:
        print(f"[ERROR] No predictions available for region {region}. Skipping.")
        continue

    actual_val = None
    if target_date in df["Month"].values:
        actual_val = df.loc[df["Month"] == target_date, target_col].values[0]

    print(f"[RESULT] Region: {region}")
    print(f"  ML Prediction:    {ml_pred:.0f}" if ml_pred is not None else "  ML Prediction: N/A")
    print(f"  LSTM Prediction:  {lstm_pred:.0f}" if lstm_pred is not None else "  LSTM Prediction: N/A")
    print(f"  Hybrid Prediction: {hybrid_pred:.0f}")
    if actual_val is not None:
        mape = mean_absolute_percentage_error([actual_val], [hybrid_pred]) * 100
        print(f"  Actual:           {actual_val:.0f}")
        print(f"  MAPE:             {mape:.2f}%")

    results.append({
        "Region": region,
        "Month": target_date,
        "ML_Prediction": int(round(ml_pred)) if ml_pred is not None else None,
        "LSTM_Prediction": int(round(lstm_pred)) if lstm_pred is not None else None,
        "Hybrid_Prediction": int(round(hybrid_pred)),
        "Actual": int(actual_val) if actual_val is not None else None,
        "MAPE": round(mape, 2) if actual_val is not None else None
    })

    # Plot 
        
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.figure(figsize=(6, 5))

    bars = []
    labels = []
    colors = []

    if actual_val is not None:
        labels.append("Actual")
        bars.append(actual_val)
        colors.append("blue")

    if ml_pred is not None:
        labels.append("ML")
        bars.append(ml_pred)
        colors.append("green")

    if lstm_pred is not None:
        labels.append("LSTM")
        bars.append(lstm_pred)
        colors.append("purple")

    labels.append("Hybrid")
    bars.append(hybrid_pred)
    colors.append("orange")

    plt.bar(labels, bars, color=colors)
    plt.title(f"{region} - Hybrid Monthly {'Validation' if actual_val is not None else 'Forecast'} ({date_str})")
    plt.ylabel("Total Attendances")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    for p in plt.gca().patches:
        plt.gca().annotate(
            f'{p.get_height():,.0f}',          
            (p.get_x() + p.get_width() / 2, p.get_height()),  
            ha='center', va='bottom', fontsize=9
        )
    plt.tight_layout()
    plt.savefig(f"forecast_plot_{region.replace(' ', '_')}_hybrid_monthly_{date_str}.png")
    plt.close()


# Save combined CSV 
results_df = pd.DataFrame(results)
results_csv_path = f"hybrid_monthly_predictions_{YEAR}_{MONTH:02d}.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\n[INFO] Saved combined predictions to {results_csv_path}")
