import pandas as pd
import os

# Config
INPUT_CSV = r"C:\Users\viraj\Downloads\nhs regional scripts\derived_weekly_regional_dataset.csv"
OUTPUT_CSV = r"C:\Users\viraj\Downloads\nhs regional scripts\regional_engineered_features_weekly.csv"

# Load and sort
df = pd.read_csv(INPUT_CSV, parse_dates=["Week"])
df = df.sort_values(["Region_unified", "Week"]).reset_index(drop=True)

# LAG FEATURES (PER REGION)
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

# ROLLING FEATURES (PER REGION)
rolling_columns = [
    "Total_Attendances",
    "Over_4hr_Waits",
    "Emergency_Admissions"
]
for col in rolling_columns:
    if col in df.columns:
        df[f"{col}_roll3_mean"] = (
            df.groupby("Region_unified")[col]
            .rolling(window=3)
            .mean()
            .reset_index(0, drop=True)
        )
        df[f"{col}_roll3_std"] = (
            df.groupby("Region_unified")[col]
            .rolling(window=3)
            .std()
            .reset_index(0, drop=True)
        )

# SEASONALITY & CALENDAR FLAGS
df["Week_Num"] = df["Week"].dt.isocalendar().week
df["Month_Num"] = df["Week"].dt.month
df["Quarter"] = df["Week"].dt.quarter
df["Is_Winter"] = df["Month_Num"].isin([11, 12, 1, 2, 3]).astype(int)
df["Is_Holiday_Month"] = df["Month_Num"].isin([12, 1]).astype(int)
df["Is_Financial_Year_End"] = (df["Month_Num"] == 3).astype(int)

# SMART FILTER (ESSENTIAL LAGS)
essential_lags = [
    "Total_Attendances_lag3",
    "Over_4hr_Waits_lag3",
    "Emergency_Admissions_lag3"
]
df_cleaned = df.dropna(subset=essential_lags).reset_index(drop=True)


# Save
df_cleaned.to_csv(OUTPUT_CSV, index=False)
print(f"Regional smart engineered WEEKLY dataset saved to:\n{OUTPUT_CSV}")
print(f"Total rows: {len(df_cleaned)}, Columns: {len(df_cleaned.columns)}")
