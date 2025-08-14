import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers import Input
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib.ticker as ticker

# Load and prepare regional data 
df = pd.read_csv("regional_engineered_features.csv", parse_dates=["Month"])
df = df.sort_values(["Region_unified", "Month"]).reset_index(drop=True)

# Assuming you collect results in a list during loop:
results = []
# Loop over regions 
regions = df["Region_unified"].unique()

# Sequences 
def create_sequences(data, target_index, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len][target_index])
    return np.array(X), np.array(y)

for region in regions:
    print(f"\n[INFO] Processing region: {region}")

    region_df = df[df["Region_unified"] == region].copy()
    region_df = region_df.drop(columns=["Region", "Region_unified"])

    target_col = "Total_Attendances"
    features = region_df.drop(columns=["Month"])
    features = features.select_dtypes(include=[np.number])

    # Scale 
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    joblib.dump(scaler, f"scaler_{region}.save")


    SEQ_LEN = 12
    target_index = features.columns.get_loc(target_col)
    X, y = create_sequences(scaled, target_index, SEQ_LEN)

    # Train/Test split 
    train_size = len(X) - 7
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Model 
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(128, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train 
    model_path = f"lstm_model_{region}.keras"
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=8,
        callbacks=[early_stop, checkpoint],
        verbose=0
    )
    model.save(model_path)
    print(f"[INFO] Explicitly saved model for {region} at {model_path}")

    # Predict 
    y_pred = model.predict(X_test)
    X_test_last = X_test[:, -1, :]
    X_pred_scaled = X_test_last.copy()
    X_pred_scaled[:, target_index] = y_pred.ravel()

    X_true_scaled = X_test_last.copy()
    X_true_scaled[:, target_index] = y_test.ravel()

    y_pred_inv = scaler.inverse_transform(X_pred_scaled)[:, target_index]
    y_test_inv = scaler.inverse_transform(X_true_scaled)[:, target_index]

    # Metrics
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    acc = 100 - mape

    print(f"""
    [INFO] Region: {region}
    [INFO] MAE: {mae:,.0f}
    [INFO] RMSE: {rmse:,.0f}
    [INFO] MAPE: {mape:.2f}%
    [INFO] Accuracy: {acc:.2f}%
    """)

    # Inside loop, after metrics calculation, add:
    results.append({
        "Region": region,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Accuracy (%)": acc
    })
    #  Save output 
    forecast_df = pd.DataFrame({
        "Region": region,
        "Actual": y_test_inv,
        "Predicted": y_pred_inv
    })
    forecast_df.to_csv(f"monthly_forecast_{region}.csv", index=False)

    # Plot 
    plt.figure(figsize=(6, 4))
    plt.plot(y_test_inv, label="Actual", marker='o')
    plt.plot(y_pred_inv, label="Predicted", marker='x')
    plt.title(f"{region} - LSTM Monthly Forecast")
    plt.xlabel("Month")
    plt.ylabel("Total Attendances")
    months = region_df["Month"].iloc[-7:].dt.strftime("%b %Y").values
    plt.xticks(ticks=range(7), labels=months, rotation=45)
    plt.legend()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    plt.tight_layout()
    plt.savefig(f"forecast_plot_{region}.png")
    plt.close()




# After the loop ends, add:
summary_df = pd.DataFrame(results)
print("[INFO] LSTM Monthly Modeling Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv("lstm_monthly_modeling_summary.csv", index=False)
