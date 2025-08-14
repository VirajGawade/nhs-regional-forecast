import argparse
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import load_model
import tensorflow as tf

# Quiet TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ==== CLI ====
parser = argparse.ArgumentParser(description="Hybrid (ML + LSTM) Weekly Prediction by Region")
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
parser.add_argument("--day", type=int, required=True)
args = parser.parse_args()

# ==== Dates: Monday start, Sunday end (matches your weekly scripts) ====
input_date = datetime(args.year, args.month, args.day)
week_start = input_date - timedelta(days=input_date.weekday())  # Monday
TARGET_DATE = week_start + timedelta(days=6)  # Sunday (stored in CSV)
iso_year, iso_week, _ = TARGET_DATE.isocalendar()
target_week_str = f"{iso_year}-W{str(iso_week).zfill(2)}"

# ==== Config paths ====
DATA_PATH = "regional_engineered_features_weekly.csv"

ML_MODEL_DIR = "regional_ml_weekly_models"
ML_SCALER_DIR = "regional_ml_weekly_scalers"

LSTM_MODEL_DIR = "."                # best_weekly_lstm_{region}.keras
LSTM_SCALER_DIR = "."               # weekly_scaler_{region}.save

OUT_DIR = "hybrid_weekly_predictions"
os.makedirs(OUT_DIR, exist_ok=True)

# ==== Hyper ====
SEQ_LEN = 12
ML_WEIGHT = 0.7
LSTM_WEIGHT = 0.3
TARGET_COL = "Total_Attendances"
REGION_COL = "Region_unified"

# ==== Load data ====
df_all = pd.read_csv(DATA_PATH, parse_dates=["Week"])
df_all = df_all.sort_values("Week")

regions = sorted(df_all[REGION_COL].dropna().unique())

results = []

def ml_week_pred(region_df, region):
    """
    Returns ML prediction (float) or None if models/scaler missing.
    """
    # Feature columns: numeric, excluding non-features
    non_feats = [TARGET_COL, "Region", REGION_COL, "Type", "Week"]
    feature_cols = [c for c in region_df.select_dtypes(include=[np.number]).columns
                    if c not in non_feats]

    # Load scaler/model
    scaler_path = os.path.join(ML_SCALER_DIR, f"weekly_scaler_{region}.save")
    rf_path = os.path.join(ML_MODEL_DIR, f"{region.replace(' ', '_')}_RandomForest_weekly.pkl")
    xgb_path = os.path.join(ML_MODEL_DIR, f"{region.replace(' ', '_')}_XGBoost_weekly.pkl")

    if not (os.path.exists(scaler_path) and os.path.exists(rf_path) and os.path.exists(xgb_path)):
        return None

    scaler = joblib.load(scaler_path)
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    # Build input features = mean of last SEQ_LEN weeks BEFORE target (validation) or last SEQ_LEN (forecast)
    if TARGET_DATE in set(region_df["Week"]):
        # validation: use rows strictly before TARGET_DATE
        input_rows = region_df[region_df["Week"] < TARGET_DATE].tail(SEQ_LEN)
    else:
        # forecast: use last SEQ_LEN rows available
        input_rows = region_df.tail(SEQ_LEN)

    if len(input_rows) < SEQ_LEN:
        return None

    mean_features = input_rows[feature_cols].mean()
    input_df = pd.DataFrame([mean_features], columns=feature_cols)

    # Align with scaler expected columns (drop target if present)
    expected_cols = list(getattr(scaler, "feature_names_in_", input_df.columns))
    if TARGET_COL in expected_cols:
        expected_cols.remove(TARGET_COL)
    input_df = input_df[expected_cols]

    X_scaled = scaler.transform(input_df)

    rf_pred = float(rf_model.predict(X_scaled)[0])
    xgb_pred = float(xgb_model.predict(X_scaled)[0])

    # Simple average (like your hybrid monthly did for ML block)
    ml_pred = (rf_pred + xgb_pred) / 2.0
    return ml_pred


def lstm_week_pred(region_df, region):
    """
    Returns LSTM prediction (float) or None if model/scaler missing.
    Uses the weekly MinMax scaler that includes the target (same as your LSTM code).
    """
    model_path = os.path.join(LSTM_MODEL_DIR, f"best_weekly_lstm_{region}.keras")
    scaler_path = os.path.join(LSTM_SCALER_DIR, f"weekly_scaler_{region}.save")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
    except Exception:
        return None

    # Scaler was fit on numeric features + target (per your weekly LSTM modelling)
    expected_features = list(scaler.feature_names_in_)
    input_features = [c for c in expected_features if c != TARGET_COL]

    # Validation: if TARGET_DATE exists, exclude it when forming the input sequence
    if TARGET_DATE in set(region_df["Week"]):
        df_hist = region_df[region_df["Week"] < TARGET_DATE].copy()
    else:
        df_hist = region_df.copy()

    # Need all expected columns
    try:
        scaled = scaler.transform(df_hist[expected_features])
    except Exception:
        return None

    if len(scaled) < SEQ_LEN:
        return None

    # Build sequence from last SEQ_LEN rows (input features only)
    X_seq = scaled[-SEQ_LEN:, [expected_features.index(c) for c in input_features]]
    X_seq = np.expand_dims(X_seq, axis=0)

    # Predict in scaled space (single value)
    pred_scaled = model.predict(X_seq, verbose=0)[0][0]

    # Inverse transform by constructing a full row with predicted target
    last_row = scaled[-1].copy()
    target_idx = expected_features.index(TARGET_COL)
    last_row[target_idx] = pred_scaled
    inv = scaler.inverse_transform([last_row])[0]
    lstm_pred = float(inv[target_idx])
    return lstm_pred


print(f"\n[INFO] Hybrid weekly prediction for week ending {TARGET_DATE.date()} (ISO {target_week_str})")

for region in regions:
    print(f"\n[INFO] Processing region: {region}")
    region_df = df_all[df_all[REGION_COL] == region].copy().sort_values("Week").reset_index(drop=True)

    if len(region_df) < SEQ_LEN + 1:
        print(f"[WARNING] Not enough history for {region}. Skipping.")
        continue

    # ---- Get ML + LSTM weekly predictions ----
    ml_pred = ml_week_pred(region_df, region)
    lstm_pred = lstm_week_pred(region_df, region)

    if (ml_pred is None) and (lstm_pred is None):
        print(f"[WARNING] No prediction available for {region} (missing models/scalers or insufficient data).")
        continue

    # Combine per weights (fallbacks if one side is missing)
    if (ml_pred is not None) and (lstm_pred is not None):
        hybrid_pred = ML_WEIGHT * ml_pred + LSTM_WEIGHT * lstm_pred
    elif ml_pred is not None:
        hybrid_pred = ml_pred
    else:
        hybrid_pred = lstm_pred

    # Actual (if validation)
    actual_val = None
    if TARGET_DATE in set(region_df["Week"]):
        actual_val = float(region_df.loc[region_df["Week"] == TARGET_DATE, TARGET_COL].values[0])

    # ---- Print nicely ----
    print(f"[RESULT] Region: {region}")
    print(f"  ML Prediction:     {ml_pred:.0f}" if ml_pred is not None else "  ML Prediction:     N/A")
    print(f"  LSTM Prediction:   {lstm_pred:.0f}" if lstm_pred is not None else "  LSTM Prediction:   N/A")
    print(f"  Hybrid Prediction: {hybrid_pred:.0f}")
    if actual_val is not None:
        mape = mean_absolute_percentage_error([actual_val], [hybrid_pred]) * 100
        print(f"  Actual:            {actual_val:.0f}")
        print(f"  MAPE:              {mape:.2f}%")

    results.append({
        "Region": region,
        "Week": TARGET_DATE.date(),
        "ML_Prediction": int(round(ml_pred)) if ml_pred is not None else None,
        "LSTM_Prediction": int(round(lstm_pred)) if lstm_pred is not None else None,
        "Hybrid_Prediction": int(round(hybrid_pred)),
        "Actual": int(round(actual_val)) if actual_val is not None else None,
        "MAPE": round(mean_absolute_percentage_error([actual_val],[hybrid_pred])*100, 2) if actual_val is not None else None
    })
    
    # === Plot for THIS region ===
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    date_str = TARGET_DATE.strftime("%Y-W%U")

    plt.figure(figsize=(5, 5))
    if actual_val is not None:  # validation run
        plt.bar(["Actual", "ML", "LSTM", "Hybrid"],
                [actual_val,
                 ml_pred if ml_pred is not None else 0,
                 lstm_pred if lstm_pred is not None else 0,
                 hybrid_pred],
                color=["blue", "green", "purple", "orange"])
        plt.title(f"{region} - Hybrid Weekly Validation ({date_str})")
    else:  # forecast run
        plt.bar(["Actual", "ML", "LSTM", "Hybrid"],
                [0,
                 ml_pred if ml_pred is not None else 0,
                 lstm_pred if lstm_pred is not None else 0,
                 hybrid_pred],
                color=["blue", "green", "purple", "orange"])
        plt.title(f"{region} - Hybrid Weekly Forecast ({date_str})")

    plt.ylabel("Total Attendances")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    plt.tight_layout()
    plt.savefig(f"forecast_plot_{region.replace(' ', '_')}_hybrid_weekly_{date_str}.png")
    plt.close()

# ==== Save combined CSV ====
out_df = pd.DataFrame(results)
out_csv = os.path.join(OUT_DIR, f"hybrid_weekly_predictions_{iso_year}_{str(iso_week).zfill(2)}.csv")
out_df.to_csv(out_csv, index=False)
print(f"\n[INFO] Saved combined predictions to {out_csv}")
