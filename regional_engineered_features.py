import pandas as pd
import os

# config
INPUT_CSV = r"C:\Users\viraj\Downloads\nhs regional scripts\final_merged_regional_dataset.csv"
OUTPUT_CSV = r"C:\Users\viraj\Downloads\nhs regional scripts\regional_engineered_features.csv"

# Load and sort
df = pd.read_csv(INPUT_CSV, parse_dates=["Month"])
df = df.sort_values(["Region_unified", "Month"]).reset_index(drop=True)

# Lag features per region
lag_columns = [
    "Total_Attendances",
    "Type1_Attendances",
    "Over_4hr_Waits",
    "Over_12hr_Waits",
    "Emergency_Admissions",
    "flu_positivity",
    "MeanTemp"
]
for col in lag_columns:
    if col in df.columns:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df.groupby("Region_unified")[col].shift(lag)

# Rolling features
rolling_columns = [
    "Total_Attendances",
    "Over_4hr_Waits",
    "Emergency_Admissions"
]
for col in rolling_columns:
    if col in df.columns:
        df[f"{col}_roll3_mean"] = df.groupby("Region_unified")[col].rolling(window=3).mean().reset_index(0, drop=True)
        df[f"{col}_roll3_std"] = df.groupby("Region_unified")[col].rolling(window=3).std().reset_index(0, drop=True)

# Seasonality 
df["Month_Num"] = df["Month"].dt.month
df["Quarter"] = df["Month"].dt.quarter
df["Is_Winter"] = df["Month_Num"].isin([11, 12, 1, 2, 3]).astype(int)
df["Is_Holiday_Month"] = df["Month_Num"].isin([12, 1]).astype(int)
df["Is_Financial_Year_End"] = (df["Month_Num"] == 3).astype(int)

# essential lags
essential_lags = [
    "Total_Attendances_lag3",
    "Over_4hr_Waits_lag3",
    "Emergency_Admissions_lag3"
]
df_cleaned = df.dropna(subset=essential_lags).reset_index(drop=True)

# Force all regions to start from July 2019 for uniformity 
df_cleaned = df_cleaned[df_cleaned["Month"] >= "2019-07-01"]

# Save
df_cleaned.to_csv(OUTPUT_CSV, index=False)
print(f" Regional smart engineered dataset saved to:\n{OUTPUT_CSV}")
print(f"Total rows: {len(df_cleaned)}, Columns: {len(df_cleaned.columns)}")
