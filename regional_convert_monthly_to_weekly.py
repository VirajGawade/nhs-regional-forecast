import pandas as pd
import numpy as np

# Load Data 
monthly_df = pd.read_csv("final_merged_regional_dataset.csv", parse_dates=["Month"])
weekly_trends = pd.read_csv("weekly_google_trends.csv", parse_dates=["date"], dayfirst=True)

# Standardize Columns 
monthly_df = monthly_df.sort_values(["Region_unified", "Month"]).reset_index(drop=True)
weekly_trends = weekly_trends.rename(columns=lambda x: x.strip())

# Identify Quantity vs Constant Columns 
quantity_cols = [
    col for col in monthly_df.columns
    if col not in ["Month", "Region", "Region_unified"] and monthly_df[col].dtype in [np.float64, np.int64]
]

constant_cols = [col for col in monthly_df.columns if col not in quantity_cols + ['Month']]

# Initialize Weekly Rows 
weekly_rows = []

# Loop Through Regions and Months 
for (region, month), group in monthly_df.groupby(["Region_unified", "Month"]):
    row = group.iloc[0]
    start = month
    end = month + pd.offsets.MonthEnd(0)

    # Filter weeks within this month
    weekly_subset = weekly_trends[(weekly_trends["date"] >= start) & (weekly_trends["date"] <= end)].copy()
    if weekly_subset.empty:
        continue

    # Compute trend weights
    weights = {}
    for trend_col in ['A&E near me', 'accident', 'ambulance', 'emergency symptoms',
                      'hospital waiting times', 'NHS 111', 'stroke', 'flu symptoms']:
        if trend_col in weekly_subset.columns:
            total = weekly_subset[trend_col].fillna(0).sum()
            w_series = weekly_subset[trend_col].fillna(0)
            w_series = w_series / total if total > 0 else pd.Series([1 / len(weekly_subset)] * len(weekly_subset))
        else:
            w_series = pd.Series([1 / len(weekly_subset)] * len(weekly_subset))
        weights[trend_col] = w_series.reset_index(drop=True)

    # Create weekly rows
    for i, week_date in enumerate(weekly_subset['date'].reset_index(drop=True)):
        new_row = {
            "Week": week_date,
            "Region": row["Region"],
            "Region_unified": row["Region_unified"]
        }

        # Weighted quantity features
        for col in quantity_cols:
            match = col.lower()
            related = next((w for w in weights if w.lower() in match), None)
            if related:
                weight = weights[related][i]
            else:
                avg_weight = np.mean([w[i] for w in weights.values()])
                weight = avg_weight
            new_row[col] = row.get(col, 0) * weight

        # Broadcast constant features
        for col in constant_cols:
            new_row[col] = row[col]

        weekly_rows.append(new_row)

# Create Final Weekly Regional DataFrame 
weekly_df = pd.DataFrame(weekly_rows)
weekly_df.sort_values(["Region_unified", "Week"], inplace=True)
weekly_df.reset_index(drop=True, inplace=True)

# Save CSV
weekly_df.to_csv("derived_weekly_regional_dataset.csv", index=False)
print(" Saved: derived_weekly_regional_dataset.csv")
