import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress Keras UserWarnings
import tensorflow as tf
import matplotlib.ticker as ticker

tf.get_logger().setLevel('ERROR')

# Load regional weekly dataset
df = pd.read_csv("regional_engineered_features_weekly.csv", parse_dates=["Week"])
df.sort_values(["Region_unified", "Week"], inplace=True)

regions = df["Region_unified"].unique()
SEQ_LEN = 12
SPLIT_DATE = pd.Timestamp("2024-11-01")

all_results = []  # Collect summary results here

for region in regions:
    print(f"\n Training weekly LSTM for region: {region}")

    region_df = df[df["Region_unified"] == region].copy()
    region_df.reset_index(drop=True, inplace=True)

    target = "Total_Attendances"
    feat_cols = [col for col in region_df.select_dtypes(include=["float64", "int64"]).columns if col != target]

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(region_df[feat_cols + [target]])

    # Ensure feature_names_in_ is correctly set including target column
    if (not hasattr(scaler, 'feature_names_in_')) or (len(scaler.feature_names_in_) != len(region_df[feat_cols + [target]].columns)):
        scaler.feature_names_in_ = region_df[feat_cols + [target]].columns.to_numpy()

    joblib.dump(scaler, f"weekly_scaler_{region}.save")

    scaled_df = pd.DataFrame(scaled, columns=feat_cols + [target])
    scaled_df["Week"] = region_df["Week"]

    # Create sequences
    def make_sequences(data, seq_len=SEQ_LEN):
        X, y, weeks = [], [], []
        for i in range(len(data) - seq_len):
            X.append(data[feat_cols].iloc[i:i + seq_len].values)
            y.append(data[target].iloc[i + seq_len])
            weeks.append(data["Week"].iloc[i + seq_len])
        return np.array(X), np.array(y), weeks

    X, y, weeks = make_sequences(scaled_df)

    # Split
    weeks = np.array(weeks)
    train_mask = weeks < SPLIT_DATE
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    weeks_test = weeks[~train_mask]

    if len(X_test) == 0:
        print(f" Skipping {region}: No test data after {SPLIT_DATE}")
        continue

    # Build model
    def build_model(units1=64, units2=32, drop=0.2):
        model = Sequential([
            LSTM(units1, return_sequences=True, input_shape=(SEQ_LEN, len(feat_cols))),
            Dropout(drop),
            LSTM(units2),
            Dropout(drop),
            Dense(1)
        ])
        return model

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
    checkpoint = ModelCheckpoint(f"best_weekly_lstm_{region}.keras", save_best_only=True, monitor="val_loss")

    def lr_schedule(epoch, lr):
        if epoch > 0 and epoch % 30 == 0:
            return lr * 0.5
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop, reduce_lr, checkpoint, lr_scheduler],
        verbose=0
    )

    # Predict and inverse-transform
    y_pred = model.predict(X_test).flatten()
    idx = len(feat_cols)
    inv_y_test = scaler.inverse_transform(np.hstack([np.zeros((len(y_test), idx)), y_test.reshape(-1,1)]))[:, idx]
    inv_y_pred = scaler.inverse_transform(np.hstack([np.zeros((len(y_pred), idx)), y_pred.reshape(-1,1)]))[:, idx]

    # Metrics
    mae = mean_absolute_error(inv_y_test, inv_y_pred)
    rmse = np.sqrt(mean_squared_error(inv_y_test, inv_y_pred))
    mape = np.mean(np.abs((inv_y_test - inv_y_pred) / inv_y_test)) * 100
    acc = 100 - mape

    print(f"""
[INFO] Region: {region}
[INFO] MAE: {mae:,.0f}
[INFO] RMSE: {rmse:,.0f}
[INFO] MAPE: {mape:.2f}%
[INFO] Accuracy: {acc:.2f}%
""")

    all_results.append({
        "Region": region,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Accuracy (%)": acc
    })

    #  Save results
    result_df = pd.DataFrame({
        "Region": region,
        "Week": weeks_test,
        "Actual": inv_y_test,
        "Predicted": inv_y_pred
    })
    result_df.to_csv(f"weekly_forecast_{region}.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(weeks_test, inv_y_test, label="Actual", marker="o")
    plt.plot(weeks_test, inv_y_pred, label="Predicted", marker="x")
    plt.title(f"{region} - Weekly LSTM Forecast")
    plt.xlabel("Week")
    plt.ylabel("Total Attendances")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    plt.tight_layout()
    plt.savefig(f"weekly_forecast_plot_{region}.png")
    plt.close()

# After loop ends: print and save summary table
summary_df = pd.DataFrame(all_results)
print("[INFO] Weekly LSTM Modeling Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv("weekly_lstm_modeling_summary.csv", index=False)
